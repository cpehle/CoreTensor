//
//  Slice.swift
//  CoreTensor
//
//  Copyright 2016-2017 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

/// Tensor slice
public struct TensorSlice<UnitType> : TensorProtocol {
    public typealias Shape = TensorShape
    public typealias BaseForm = Tensor<UnitType>

    public private(set) var base: BaseForm
    private var baseIndices: [Int]
    private var bounds: CountableRange<Int>?

    private init(base: Tensor<UnitType>, baseIndices indices: [Int], bounds: CountableRange<Int>?) {
        precondition(zip(base.shape, indices).forAll { $1 >= 0 && $1 < $0 },
                     "Element indices are out of bounds")
        self.base = base
        self.baseIndices = indices
        self.bounds = bounds
    }
}

public extension TensorSlice {
    init(_ base: Tensor<UnitType>) {
        self.init(base: base, baseIndices: [], bounds: nil)
    }

    init(base: Tensor<UnitType>, bounds: CountableRange<Int>?) {
        self.init(base: base, baseIndices: [], bounds: bounds)
    }

    init(base: TensorSlice, bounds: CountableRange<Int>?) {
        if let bounds = bounds {
            precondition(base.indices.contains(bounds.startIndex)
                && base.indices.contains(bounds.endIndex - 1),
                         "Slice is out of bounds")
        }
        self.init(base: base.base, baseIndices: base.baseIndices, bounds: bounds)
    }

    init(base: Tensor<UnitType>, indices: [Int]) {
        self.init(base: base, baseIndices: indices, bounds: nil)
    }

    init(base: TensorSlice, indices: [Int]) {
        self.init(base: base.base, baseIndices: base.baseIndices + indices, bounds: nil)
    }

    init(base: Tensor<UnitType>, index: Int) {
        self.init(base: base, indices: [index])
    }

    init(base: TensorSlice, index: Int) {
        self.init(base: base.base, indices: base.baseIndices + [index])
    }

    init(_ base: TensorSlice<UnitType>) {
        self.init(base: base, bounds: base.indices)
    }

    internal init(elementShape: TensorShape?, units: ContiguousArray<UnitType>) {
        self.init(Tensor(elementShape: elementShape, units: units))
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    fileprivate init(shape: TensorShape, units: ContiguousArray<UnitType>) {
        self.init(Tensor(shape: shape, units: units))
    }

    /// Initialize an empty tensor of scalar elements
    init() {
        self.init(Tensor())
    }

    /// Initialize an empty tensor
    init(elementShape: TensorShape) {
        self.init(Tensor(elementShape: elementShape))
    }

    /// Initialize a scalar tensor
    init(scalar: UnitType) {
        self.init(Tensor(scalar: scalar))
    }

    /// Initialize a tensor from a sequence of tensors of element shape
    init<S: Sequence>(elementShape: TensorShape, elements: S)
        where S.Element : TensorProtocol, S.Element.UnitType == UnitType
    {
        self.init(Tensor(elementShape: elementShape, elements: elements))
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    init<S: Sequence>(shape: TensorShape, units: S,
                             vacancySupplier supplier: (() -> UnitType)? = nil)
        where S.Iterator.Element == UnitType
    {
        self.init(Tensor(shape: shape, units: units, vacancySupplier: supplier))
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    init(shape: TensorShape, repeating repeatedValue: UnitType) {
        self.init(Tensor(shape: shape, repeating: repeatedValue))
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    init(shape: TensorShape, supplier: () -> UnitType) {
        self.init(Tensor(shape: shape, supplier: supplier))
    }
}

public extension TensorSlice {
    /// Indexing depth of this slice, i.e. the rank difference between the base
    /// and the slice
    private var indexingDepth: Int {
        return baseIndices.count
    }

    /// Element tensor shape
    var elementShape: TensorShape? {
        return base.elementShape.flatMap { baseElemShape in
            if baseElemShape.rank + 1 == indexingDepth {
                return nil
            }
            return baseElemShape.dropFirst(indexingDepth)
        }
    }

    var isScalar: Bool {
        return indexingDepth == base.shape.count
    }

    /// The number of units per element
    var unitCountPerElement: Int {
        return elementShape?.contiguousSize ?? 0
    }

    var unitRange: CountableRange<Int> {
        let trimmedShape = base.shape.dropFirst()
        var (start, end) = baseIndices.enumerated().reduce((0, base.unitCount), { (acc, next) in
            let stride = trimmedShape.dropFirst(next.offset).reduce(1, *)
            if next.offset == indexingDepth - 1 {
                let temp = acc.0 + next.element * stride
                return (temp, temp + stride)
            }
            return (acc.0 + next.element * stride, acc.1)
        })
        if let bounds = bounds {
            let stride = trimmedShape.dropFirst(indexingDepth).reduce(1, *)
            (start, end) = (start + bounds.startIndex * stride,
                            start + bounds.endIndex * stride)
        }
        return start..<end
    }

    var units: ArraySlice<UnitType> {
        get {
            return base.units[unitRange]
        }
        set(newUnits) {
            base.units[unitRange] = newUnits
        }
    }

    var indices: CountableRange<Int> {
        if let bounds = bounds {
            return bounds
        } else if indexingDepth < base.shape.rank {
            return 0..<(base.shape[indexingDepth])
        }
        return 0..<1
    }

    var startIndex: Int {
        return indices.startIndex
    }

    var endIndex: Int {
        return indices.endIndex
    }

    /// Capacity reserved for element tensors
    var capacity: Int {
        return units.capacity / unitCountPerElement
    }
}

public extension TensorSlice where UnitType : Strideable, UnitType.Stride : SignedInteger {
    init(scalarElementsIn bounds: CountableRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }

    init(scalarElementsIn bounds: CountableClosedRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }
}

public extension TensorSlice {
    func makeUnitIterator() -> IndexingIterator<ArraySlice<UnitType>> {
        return units.makeIterator()
    }

    mutating func updateUnit(at index: Int, to newValue: UnitType) {
        base.updateUnit(at: index, to: newValue)
    }
}

public extension TensorSlice where UnitType : Numeric {
    mutating func incrementUnit(at index: Int, by newValue: UnitType) {
        base.incrementUnit(at: index, by: newValue)
    }

    mutating func decrementUnit(at index: Int, by newValue: UnitType) {
        base.decrementUnit(at: index, by: newValue)
    }

    mutating func multiplyUnit(at index: Int, by newValue: UnitType) {
        base.multiplyUnit(at: index, by: newValue)
    }
}

public extension TensorSlice where UnitType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        base.divideUnit(at: index, by: newValue)
    }
}

public extension TensorSlice where UnitType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        base.divideUnit(at: index, by: newValue)
    }
}

extension TensorSlice : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = TensorSlice<UnitType>
    public typealias SubSequence = TensorSlice<UnitType>

    /// Access the element tensor specified by a TensorIndex
    public subscript(index: TensorIndex) -> TensorSlice<UnitType> {
        get {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return TensorSlice(base: self, bounds: range)
        }
        set {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            base.units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access the element tensor specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the raw rank of the tensor
    public subscript(indices: Int...) -> TensorSlice<UnitType> {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access the element tensor in the current dimension at an index
    public subscript(index: Int) -> Element {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            return TensorSlice(base: self, index: index)
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = unitIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            base.units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access the subtensor specified by a contiguous range of indices
    public subscript(bounds: Range<Int>) -> SubSequence {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            return TensorSlice(base: self, bounds: CountableRange(bounds))
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            precondition(newValue.shape == elementShape?.prepending(bounds.count),
                         "Shape mismatch")
            base.units[unitSubrange(from: CountableRange(bounds))] =
                newValue.base.units[newValue.unitSubrange(from: newValue.indices)]
        }
    }
}

public extension TensorSlice {
    func reshaped(as newShape: TensorShape) -> Tensor<UnitType>? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, units: units)
    }
}

public extension TensorSlice {
    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeBufferPointer { ptr in
            try body(ptr)
        }
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeMutableBufferPointer { ptr in
            try body(&ptr)
        }
    }
}

extension TensorSlice : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write(
            isScalar ? String(describing: units.first!)
                : "[\(map {"\($0)"}.joined(separator: ", "))]"
        )
    }
}
